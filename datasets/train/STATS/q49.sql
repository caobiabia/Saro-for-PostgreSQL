select  count(*) from votes as v,  		posts as p,  		badges as b,         users as u  where u.Id = v.UserId 	and u.Id = p.OwnerUserId 	and u.Id = b.UserId  AND b.Date>='2010-07-26 19:18:41'::timestamp  AND p.Score>=-1  AND p.Score<=14  AND p.AnswerCount<=8  AND p.CommentCount<=14  AND u.Views<=38  AND u.DownVotes>=0;