select  count(*) from comments as c,  		posts as p,          postLinks as pl,          votes as v,          badges as b,          users as u  where p.Id = c.PostId 	and p.Id = pl.RelatedPostId     and p.Id = v.PostId  	and u.Id = p.LastEditorUserId     and u.Id = b.UserId  AND c.Score=0  AND c.CreationDate>='2010-08-18 02:58:40'::timestamp  AND pl.LinkTypeId=1  AND p.PostTypeId=1  AND p.Score>=-4  AND p.Score<=20  AND p.AnswerCount>=0  AND p.CommentCount>=0  AND p.CommentCount<=24  AND u.Reputation>=1  AND u.Reputation<=165  AND u.UpVotes>=0  AND u.UpVotes<=13;