select  count(*) from votes as v,  		posts as p,  		badges as b,         users as u  where u.Id = v.UserId 	and u.Id = p.OwnerUserId 	and u.Id = b.UserId  AND p.AnswerCount>=0  AND p.CreationDate>='2010-09-24 09:19:30'::timestamp  AND u.Views>=0  AND u.DownVotes<=3  AND u.UpVotes<=79  AND v.VoteTypeId=2;