select  count(*) from comments as c,  		posts as p,          postLinks as pl,          votes as v,          badges as b,          users as u  where p.Id = c.PostId 	and p.Id = pl.RelatedPostId     and p.Id = v.PostId  	and u.Id = p.LastEditorUserId     and u.Id = b.UserId  AND pl.CreationDate>='2010-10-19 15:38:38'::timestamp  AND p.Score>=0  AND p.Score<=45  AND p.ViewCount>=0  AND p.ViewCount<=1919  AND p.AnswerCount<=6  AND p.CommentCount<=11  AND p.CreationDate<='2014-08-22 11:58:07'::timestamp  AND u.Reputation<=1537  AND u.UpVotes>=0  AND u.UpVotes<=23  AND u.CreationDate>='2010-08-20 20:37:32'::timestamp  AND u.CreationDate<='2014-09-05 15:21:39'::timestamp  AND v.VoteTypeId=2  AND v.CreationDate>='2010-07-20 00:00:00'::timestamp;